"""
Explore Textract output on a real loan PDF.

Usage:
    python scripts/test_textract.py path/to/loan.pdf

What this does:
  1. Uploads the PDF to S3 under textract-tests/{filename}
  2. Runs Textract (sync for 1-page, async for multi-page)
  3. Prints a readable breakdown of what Textract found:
     - Per-page text
     - Confidence scores
     - Handwriting flags
     - Tables (if any)
     - Block type summary
  4. Saves full raw JSON response to scripts/output/{filename}_raw.json
     so you can inspect everything Textract returned

Nothing permanent — test prefix is separate from real session data.
"""
import sys
import os
import json
import time
import tempfile
from pathlib import Path

# Allow imports from fastapi_service
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "fastapi_service"))

import boto3
import fitz  # PyMuPDF — only for page count
from botocore.exceptions import ClientError
from dotenv import load_dotenv

load_dotenv()

# ------------------------------------------------------------------
# Config from env
# ------------------------------------------------------------------
BUCKET = os.environ["S3_BUCKET"]
REGION = os.environ.get("AWS_REGION", "us-east-1")
AWS_KEY = os.environ["AWS_ACCESS_KEY_ID"]
AWS_SECRET = os.environ["AWS_SECRET_ACCESS_KEY"]

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


# ------------------------------------------------------------------
# Upload
# ------------------------------------------------------------------

def upload_pdf(s3, local_path: Path) -> str:
    key = f"textract-tests/{local_path.name}"
    print(f"\n[1/4] Uploading {local_path.name} → s3://{BUCKET}/{key}")
    s3.upload_file(str(local_path), BUCKET, key)
    print(f"      Done.")
    return key


# ------------------------------------------------------------------
# Page count
# ------------------------------------------------------------------

def get_page_count(local_path: Path) -> int:
    doc = fitz.open(str(local_path))
    count = doc.page_count
    doc.close()
    return count


# ------------------------------------------------------------------
# Sync extraction (1-page)
# ------------------------------------------------------------------

def run_sync(textract, s3_key: str) -> dict:
    print(f"\n[2/4] Running Textract sync (single page)...")
    response = textract.detect_document_text(
        Document={"S3Object": {"Bucket": BUCKET, "Name": s3_key}}
    )
    print(f"      Done. {len(response.get('Blocks', []))} blocks returned.")
    return response


# ------------------------------------------------------------------
# Async extraction (multi-page)
# ------------------------------------------------------------------

def run_async(textract, s3_key: str) -> dict:
    print(f"\n[2/4] Starting Textract async job (multi-page)...")
    start = textract.start_document_text_detection(
        DocumentLocation={"S3Object": {"Bucket": BUCKET, "Name": s3_key}}
    )
    job_id = start["JobId"]
    print(f"      Job ID: {job_id}")
    print(f"      Polling every {POLL_INTERVAL}s...")

    for attempt in range(1, MAX_POLLS + 1):
        resp = textract.get_document_text_detection(JobId=job_id)
        status = resp["JobStatus"]
        print(f"      [{attempt}] Status: {status}")

        if status == "SUCCEEDED":
            all_blocks = list(resp.get("Blocks", []))
            next_token = resp.get("NextToken")
            while next_token:
                page_resp = textract.get_document_text_detection(
                    JobId=job_id, NextToken=next_token
                )
                all_blocks.extend(page_resp.get("Blocks", []))
                next_token = page_resp.get("NextToken")
            resp["Blocks"] = all_blocks
            print(f"      Complete. {len(all_blocks)} total blocks.")
            return resp

        elif status == "FAILED":
            print(f"      FAILED: {resp.get('StatusMessage')}")
            sys.exit(1)

        time.sleep(POLL_INTERVAL)

    print("Timed out.")
    sys.exit(1)


# ------------------------------------------------------------------
# Analysis + pretty print
# ------------------------------------------------------------------

def analyse_and_print(response: dict, filename: str):
    blocks = response.get("Blocks", [])

    # Block type counts
    type_counts: dict[str, int] = {}
    for b in blocks:
        t = b.get("BlockType", "UNKNOWN")
        type_counts[t] = type_counts.get(t, 0) + 1

    print(f"\n[3/4] Block type summary:")
    for btype, count in sorted(type_counts.items()):
        print(f"      {btype:<30} {count}")

    # Per-page breakdown
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
    print(f"\n[4/4] Per-page extraction:")
    print(f"      Total pages detected: {len(all_pages)}\n")

    for page in all_pages:
        lines = page_lines.get(page, [])
        confs = page_word_conf.get(page, [])
        hw_words = page_handwriting.get(page, [])

        avg_conf = sum(confs) / len(confs) if confs else 0.0
        low_conf = [c for c in confs if c < 80]

        print(f"  {'='*60}")
        print(f"  PAGE {page}")
        print(f"  {'='*60}")
        print(f"  Words: {len(confs)}  |  Avg confidence: {avg_conf:.1f}%  |  Low-conf words (<80%): {len(low_conf)}")
        if hw_words:
            print(f"  HANDWRITING DETECTED: {hw_words}")
        else:
            print(f"  Handwriting: none")
        print(f"\n  --- Extracted text ---")
        for line in lines:
            print(f"  {line}")
        print()

    # Save raw JSON
    out_path = OUTPUT_DIR / f"{Path(filename).stem}_raw.json"
    with open(out_path, "w") as f:
        json.dump(response, f, indent=2, default=str)
    print(f"\nFull raw Textract JSON saved to: {out_path}")
    print("Open it to see every block, bounding box, confidence score, and relationship.")


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/test_textract.py path/to/document.pdf")
        sys.exit(1)

    local_path = Path(sys.argv[1]).resolve()
    if not local_path.exists():
        print(f"File not found: {local_path}")
        sys.exit(1)

    if local_path.suffix.lower() != ".pdf":
        print("Only PDF files supported.")
        sys.exit(1)

    s3, textract = make_clients()

    page_count = get_page_count(local_path)
    print(f"PDF: {local_path.name}  |  Pages: {page_count}")

    s3_key = upload_pdf(s3, local_path)

    if page_count == 1:
        response = run_sync(textract, s3_key)
    else:
        response = run_async(textract, s3_key)

    analyse_and_print(response, local_path.name)


if __name__ == "__main__":
    main()
