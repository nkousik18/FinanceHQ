"""
Fetch the corpus from the manifest (scope v2 §5.0, F9, F11).

    # 1. (Re)build the manifest from the approved EDGAR selection + CFPB form list
    venv/bin/python -m evals.corpus.fetch build-manifest
    # 2. Download, verify, render. --freeze records checksums + page counts the first time.
    venv/bin/python -m evals.corpus.fetch fetch --freeze

Outputs (gitignored): data/corpus/source/<doc_id>.html, data/corpus/pdf/<doc_id>.pdf

CFPB forms can't be downloaded by script (consumerfinance.gov returns 403);
put the PDFs in data/corpus/inbox/cfpb/ by hand. `fetch` lists any missing.
Idempotent: an existing PDF is kept unless --force. A source whose sha256
differs from the frozen value is an error (the filing changed or the
download is corrupt).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path

import shutil

from evals.corpus import cfpb
from evals.corpus.manifest import DocRow, build_manifest, read_manifest, write_manifest
from evals.corpus.render import chrome_version, pdf_page_count, render_fitted
from evals.corpus.sec_client import SECClient

log = logging.getLogger("evals.corpus.fetch")

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "data" / "corpus"


class ChecksumError(RuntimeError):
    pass


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def verify_or_freeze(row: DocRow, data: bytes, freeze: bool) -> None:
    digest = sha256(data)
    if row.source_sha256:
        if row.source_sha256 != digest:
            raise ChecksumError(f"{row.doc_id}: sha256 {digest[:12]} != manifest {row.source_sha256[:12]}")
    elif freeze:
        row.source_sha256 = digest
    else:
        raise ChecksumError(f"{row.doc_id}: no frozen checksum; run with --freeze once")


def fetch_edgar(rows: list[DocRow], sec: SECClient, freeze: bool, force: bool) -> dict:
    renderer = chrome_version()
    stats = {"fetched": 0, "rendered": 0, "skipped": 0, "images_removed": 0}
    for row in rows:
        if row.part != "edgar":
            continue
        src = DATA / "source" / f"{row.doc_id}.html"
        pdf = DATA / "pdf" / f"{row.doc_id}.pdf"

        data = sec.get_bytes(row.source_url)
        verify_or_freeze(row, data, freeze)
        src.parent.mkdir(parents=True, exist_ok=True)
        src.write_bytes(data)
        stats["fetched"] += 1

        if pdf.exists() and not force:
            stats["skipped"] += 1
        else:
            r = render_fitted(data, pdf)
            stats["images_removed"] += r["images_removed"]
            stats["rendered"] += 1
            if not r["fitted"]:
                stats.setdefault("not_fitted", []).append(row.doc_id)
                log.warning("doc=%s pages=%d > source_pages=%d even at zoom %s",
                            row.doc_id, r["pages"], r["source_pages"], r["zoom"])
            if freeze:
                row.renderer = renderer
                row.render_zoom = str(r["zoom"])
                row.source_pages = str(r["source_pages"])
                row.pdf_pages = ""            # re-frozen below from the new PDF

        pages = pdf_page_count(pdf)
        if freeze and not row.pdf_pages:
            row.pdf_pages = str(pages)
        log.info("doc=%s pages=%d", row.doc_id, pages)
    return stats


def import_consumer(rows: list[DocRow], freeze: bool) -> dict:
    """Copy hand-downloaded CFPB PDFs from the inbox, verifying their checksums."""
    stats = {"imported": 0, "missing": []}
    for row in rows:
        if row.part != "consumer":
            continue
        src = cfpb.inbox_path(row)
        if not src.exists():
            stats["missing"].append(row.source_url)
            continue
        data = src.read_bytes()
        verify_or_freeze(row, data, freeze)
        pdf = DATA / "pdf" / f"{row.doc_id}.pdf"
        pdf.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, pdf)
        if freeze and not row.pdf_pages:
            row.pdf_pages = str(pdf_page_count(pdf))
        stats["imported"] += 1
    if stats["missing"]:
        log.warning("%d CFPB PDFs missing; download them into %s:\n  %s",
                    len(stats["missing"]), cfpb.INBOX, "\n  ".join(stats["missing"]))
    return stats


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build-manifest")
    f = sub.add_parser("fetch")
    f.add_argument("--freeze", action="store_true", help="record checksums/page counts not yet in the manifest")
    f.add_argument("--force", action="store_true", help="re-render PDFs that already exist")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.cmd == "build-manifest":
        rows = build_manifest()
        log.info("manifest rows=%d", len(rows))
        return 0

    rows = read_manifest()
    if not rows:
        log.error("empty manifest; run build-manifest first")
        return 1
    stats = fetch_edgar(rows, SECClient(), freeze=args.freeze, force=args.force)
    stats["consumer"] = import_consumer(rows, freeze=args.freeze)
    if args.freeze:
        write_manifest(rows)
    pages = {part: sum(int(r.pdf_pages or 0) for r in rows if r.part == part)
             for part in ("edgar", "consumer")}
    print(json.dumps({**stats, "pages": pages}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
