"""
Run cleaning + validation + markdown generation on the saved Textract JSON.
Does NOT call AWS — reads from scripts/output/ locally.

Usage:
    python scripts/test_clean_validate.py scripts/output/loan1_raw.json

Outputs:
    scripts/output/loan1_cleaned.md
"""
import sys
import os
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "fastapi_service"))

from dotenv import load_dotenv
from app.core.logging import setup_logging, get_logger
from app.pipeline.extractor import (
    ExtractionResult, PageResult, FormField, Table, TableCell,
    _assemble_result,
)
from app.pipeline.cleaner import clean_extraction
from app.pipeline.validator import validate_extraction
from app.pipeline.markdown_writer import write_markdown

load_dotenv()
setup_logging(log_level=os.environ.get("LOG_LEVEL", "INFO"), environment="development")
logger = get_logger("test_clean_validate")

OUTPUT_DIR = Path(__file__).parent / "output"


def load_result_from_json(json_path: Path, session_id: str = "test-session") -> ExtractionResult:
    """Reconstruct ExtractionResult from saved raw Textract JSON — no AWS call needed."""
    logger.info("loading_textract_json", path=str(json_path))
    with open(json_path) as f:
        data = json.load(f)

    blocks = data.get("Blocks", [])
    job_id = data.get("JobId")
    logger.info("blocks_loaded", count=len(blocks))

    return _assemble_result(session_id=session_id, blocks=blocks, job_id=job_id)


def main():
    if len(sys.argv) < 2:
        logger.error("usage", msg="python scripts/test_clean_validate.py scripts/output/loan1_raw.json")
        sys.exit(1)

    json_path = Path(sys.argv[1]).resolve()
    if not json_path.exists():
        logger.error("file_not_found", path=str(json_path))
        sys.exit(1)

    session_id = json_path.stem.replace("_raw", "")

    # Step 1: Reconstruct extraction result from saved JSON
    extraction = load_result_from_json(json_path, session_id=session_id)
    logger.info(
        "extraction_loaded",
        pages=extraction.total_pages,
        form_fields=len(extraction.form_fields),
        tables=len(extraction.tables),
        confidence=extraction.overall_avg_confidence,
        handwriting=extraction.has_handwriting,
    )

    # Step 2: Clean
    cleaned = clean_extraction(extraction)

    # Step 3: Validate
    report = validate_extraction(cleaned)
    logger.info("validation_summary", summary=report.summary())

    # Step 4: Generate markdown
    md = write_markdown(cleaned, report)

    # Save
    out_path = OUTPUT_DIR / f"{session_id}_cleaned.md"
    out_path.write_text(md, encoding="utf-8")
    logger.info("markdown_saved", path=str(out_path), size_kb=round(len(md) / 1024, 1))

    # Print quick summary to console
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"  Pages:            {cleaned.total_pages}")
    print(f"  Form fields:      {len(cleaned.form_fields)} (was {len(extraction.form_fields)})")
    print(f"  Tables:           {len(cleaned.tables)}")
    print(f"  Confidence:       {cleaned.overall_avg_confidence:.1f}%")
    print(f"  Handwriting:      {cleaned.has_handwriting}")
    print(f"  Validation:       {'PASS' if report.passed else 'FAIL'}")
    print(f"  Errors:           {len(report.errors)}")
    print(f"  Warnings:         {len(report.warnings)}")
    print(f"  Required found:   {len(report.required_fields_found)}/{len(report.required_fields_found) + len(report.required_fields_missing)}")
    print(f"  Output:           {out_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
