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

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "fastapi_service"))

import boto3
import fitz
from botocore.exceptions import ClientError
from dotenv import load_dotenv

load_dotenv()

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
    print(f"\n[1/4] Uploading {local_path.name} → s3://{BUCKET}/{key}")
    s3.upload_file(str(local_path), BUCKET, key)
    print(f"      Done.")
    return key


def run_sync(textract, s3_key: str) -> dict:
    print(f"\n[2/4] Running AnalyzeDocument sync (TABLES + FORMS)...")
    response = textract.analyze_document(
        Document={"S3Object": {"Bucket": BUCKET, "Name": s3_key}},
        FeatureTypes=FEATURE_TYPES,
    )
    print(f"      Done. {len(response.get('Blocks', []))} blocks returned.")
    return response


def run_async(textract, s3_key: str) -> dict:
    print(f"\n[2/4] Starting StartDocumentAnalysis async (TABLES + FORMS)...")
    start = textract.start_document_analysis(
        DocumentLocation={"S3Object": {"Bucket": BUCKET, "Name": s3_key}},
        FeatureTypes=FEATURE_TYPES,
    )
    job_id = start["JobId"]
    print(f"      Job ID: {job_id}")
    print(f"      Polling every {POLL_INTERVAL}s...")

    for attempt in range(1, MAX_POLLS + 1):
        resp = textract.get_document_analysis(JobId=job_id)
        status = resp["JobStatus"]
        print(f"      [{attempt}] Status: {status}")

        if status == "SUCCEEDED":
            all_blocks = list(resp.get("Blocks", []))
            next_token = resp.get("NextToken")
            while next_token:
                page_resp = textract.get_document_analysis(JobId=job_id, NextToken=next_token)
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


def analyse_and_print(response: dict, filename: str):
    blocks = response.get("Blocks", [])
    block_map = {b["Id"]: b for b in blocks}

    # Block type summary
    type_counts: dict[str, int] = {}
    for b in blocks:
        t = b.get("BlockType", "UNKNOWN")
        type_counts[t] = type_counts.get(t, 0) + 1

    print(f"\n[3/4] Block type summary:")
    for btype, count in sorted(type_counts.items()):
        print(f"      {btype:<30} {count}")

    # Per-page text + confidence
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
    print(f"      Total pages: {len(all_pages)}\n")

    for page in all_pages:
        lines = page_lines.get(page, [])
        confs = page_word_conf.get(page, [])
        hw = page_handwriting.get(page, [])
        avg = sum(confs) / len(confs) if confs else 0.0
        low = sum(1 for c in confs if c < 80)

        print(f"  {'='*60}")
        print(f"  PAGE {page}")
        print(f"  {'='*60}")
        print(f"  Words:{len(confs)}  Avg conf:{avg:.1f}%  Low-conf:{low}  HW words:{len(hw)}")
        if hw:
            print(f"  Handwriting: {hw[:8]}")
        print(f"\n  Text (first 8 lines):")
        for line in lines[:8]:
            print(f"    {line}")

    # Form fields
    print(f"\n{'='*60}")
    print(f"FORM FIELDS (KEY → VALUE)")
    print(f"{'='*60}")
    kv_count = 0
    for b in blocks:
        if b.get("BlockType") != "KEY_VALUE_SET":
            continue
        if "KEY" not in b.get("EntityTypes", []):
            continue

        # Key text
        key_word_ids = [
            wid for rel in b.get("Relationships", [])
            if rel["Type"] == "CHILD" for wid in rel["Ids"]
        ]
        key_text = " ".join(
            block_map[wid].get("Text", "") for wid in key_word_ids
            if wid in block_map and block_map[wid].get("BlockType") == "WORD"
        ).strip()

        # Value text
        val_ids = [
            wid for rel in b.get("Relationships", [])
            if rel["Type"] == "VALUE" for wid in rel["Ids"]
        ]
        value_text = ""
        for val_id in val_ids:
            val = block_map.get(val_id)
            if not val:
                continue
            vword_ids = [
                wid for rel in val.get("Relationships", [])
                if rel["Type"] == "CHILD" for wid in rel["Ids"]
            ]
            value_text = " ".join(
                block_map[wid].get("Text", "") for wid in vword_ids
                if wid in block_map and block_map[wid].get("BlockType") == "WORD"
            ).strip()

        if key_text:
            page = b.get("Page", "?")
            print(f"  [p{page}] {key_text:<40} → {value_text}")
            kv_count += 1

    print(f"\n  Total form fields: {kv_count}")

    # Tables
    print(f"\n{'='*60}")
    print(f"TABLES")
    print(f"{'='*60}")
    table_count = 0
    for b in blocks:
        if b.get("BlockType") != "TABLE":
            continue
        table_count += 1
        page = b.get("Page", "?")
        cell_ids = [
            wid for rel in b.get("Relationships", [])
            if rel["Type"] == "CHILD" for wid in rel["Ids"]
        ]
        max_row = max_col = 0
        cells = []
        for cid in cell_ids:
            cell = block_map.get(cid)
            if not cell or cell.get("BlockType") != "CELL":
                continue
            r = cell.get("RowIndex", 0)
            c = cell.get("ColumnIndex", 0)
            max_row = max(max_row, r)
            max_col = max(max_col, c)
            wids = [
                wid for rel in cell.get("Relationships", [])
                if rel["Type"] == "CHILD" for wid in rel["Ids"]
            ]
            text = " ".join(
                block_map[wid].get("Text", "") for wid in wids
                if wid in block_map and block_map[wid].get("BlockType") == "WORD"
            )
            cells.append((r, c, text))

        grid = {(r, c): t for r, c, t in cells}
        print(f"\n  Table {table_count} — Page {page} ({max_row} rows x {max_col} cols):")
        for r in range(1, min(max_row + 1, 6)):  # print first 5 rows
            row_vals = [grid.get((r, c), "") for c in range(1, max_col + 1)]
            print(f"    {' | '.join(row_vals)}")
        if max_row > 5:
            print(f"    ... ({max_row - 5} more rows)")

    if table_count == 0:
        print("  No tables detected.")

    # Save raw JSON
    out_path = OUTPUT_DIR / f"{Path(filename).stem}_raw.json"
    with open(out_path, "w") as f:
        json.dump(response, f, indent=2, default=str)
    print(f"\nFull raw JSON saved to: {out_path}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/test_textract.py path/to/document.pdf")
        sys.exit(1)

    local_path = Path(sys.argv[1]).resolve()
    if not local_path.exists():
        print(f"File not found: {local_path}")
        sys.exit(1)

    s3, textract = make_clients()
    page_count = get_page_count(local_path)
    print(f"PDF: {local_path.name}  |  Pages: {page_count}")

    s3_key = upload_pdf(s3, local_path)

    response = run_sync(textract, s3_key) if page_count == 1 else run_async(textract, s3_key)
    analyse_and_print(response, local_path.name)


if __name__ == "__main__":
    main()
