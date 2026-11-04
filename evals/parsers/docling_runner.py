"""
Parse the corpus with Docling and cache the results (scope v2 §6.0, D6, S0).

RUNS IN venv-docling (Docling's numpy/torch/httpx versions conflict with the
app venv). The cached JSON is read by the rest of the evals from the app venv.

    venv-docling/bin/python -m evals.parsers.docling_runner --configs P1 P2
    # long run, unattended, Mac kept awake:
    caffeinate -i venv-docling/bin/python -m evals.parsers.docling_runner --configs P1 P2

Resumable: documents already cached with the same cache key are skipped.
Each cache file holds:
  meta        doc id, source sha256, parser + versions, config, cache key, timings
  confidence  Docling's per-page and document confidence report
  ocr_cells   per page: OCR'd text cells with bbox and confidence (OCR pages only)
  document    DoclingDocument.export_to_dict(): layout items, labels, bboxes,
              tables with cells, reading order (lossless; metrics read from here)

D6 time cap: the run projects total time after each document and stops if the
projection exceeds --max-hours (default 8), so a slow run asks before burning the night.
"""
from __future__ import annotations

import argparse
import importlib.metadata as md
import json
import logging
import math
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from evals.corpus.manifest import read_manifest  # noqa: E402  (stdlib only)
from evals.parsers.config import (  # noqa: E402
    CACHE_ROOT, CONFIGS, ParserConfig, cache_key, cache_path, is_cached,
)

log = logging.getLogger("evals.parsers.docling")
PDF_DIR = REPO / "data" / "corpus" / "pdf"


def versions() -> dict:
    v = {p: md.version(p) for p in ("docling", "docling-core", "docling-ibm-models", "docling-parse", "torch")}
    try:
        out = subprocess.run(["tesseract", "--version"], capture_output=True, text=True, check=True)
        v["tesseract"] = out.stdout.split()[1]
    except Exception:
        v["tesseract"] = "unavailable"
    return v


def build_converter(cfg: ParserConfig):
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import (
        AcceleratorDevice, AcceleratorOptions, PdfPipelineOptions, TableFormerMode,
        TesseractCliOcrOptions,
    )
    from docling.document_converter import DocumentConverter, PdfFormatOption

    if cfg.ocr_engine != "tesseract_cli":
        raise ValueError(f"unsupported ocr_engine {cfg.ocr_engine}")
    opts = PdfPipelineOptions(
        do_ocr=True,
        do_table_structure=True,
        ocr_options=TesseractCliOcrOptions(lang=[cfg.ocr_lang]),
        generate_parsed_pages=True,          # keeps OCR cells with confidence
    )
    opts.table_structure_options.mode = TableFormerMode(cfg.table_mode)
    opts.accelerator_options = AcceleratorOptions(
        device=AcceleratorDevice(cfg.device), num_threads=cfg.num_threads)
    return DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)})


def _num(x):
    """JSON-safe float (Docling uses NaN for 'not applicable')."""
    return None if isinstance(x, float) and math.isnan(x) else x


def extract(result) -> dict:
    conf = result.confidence
    pages = {}
    for page_no, pc in conf.pages.items():
        d = pc.model_dump(mode="json")
        pages[str(page_no)] = {k: _num(v) for k, v in d.items()}
    confidence = {k: _num(v) for k, v in conf.model_dump(mode="json").items() if k != "pages"}
    confidence["pages"] = pages

    ocr_cells = {}
    for page in result.pages:
        pp = page.parsed_page
        if pp is None:
            continue
        cells = [
            {"text": c.text, "confidence": round(float(c.confidence), 4),
             "bbox": [round(v, 2) for v in c.rect.to_bounding_box().as_tuple()]}
            for c in pp.textline_cells if getattr(c, "from_ocr", False)
        ]
        if cells:
            ocr_cells[str(page.page_no)] = cells

    return {"confidence": confidence, "ocr_cells": ocr_cells,
            "document": result.document.export_to_dict(),
            "status": result.status.value,
            "errors": [str(e) for e in result.errors]}


def parse_one(converter, cfg: ParserConfig, row, vers: dict, key: str) -> dict:
    pdf = PDF_DIR / f"{row.doc_id}.pdf"
    t0 = time.monotonic()
    result = converter.convert(pdf)
    seconds = round(time.monotonic() - t0, 2)
    meta = {
        "cache_key": key,                    # first key in the file (see is_cached)
        "doc_id": row.doc_id, "part": row.part, "session": row.deal_id,
        "source_sha256": row.source_sha256, "pages": int(row.pdf_pages or 0),
        "parser": cfg.parser, "versions": vers, "config": cfg.as_dict(),
        "seconds": seconds, "parsed_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    return {"meta": meta, **extract(result)}


def write_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload))
    os.replace(tmp, path)


def run(config_names: list[str], max_hours: float, only: list[str] | None, limit: int | None) -> dict:
    rows = [r for r in read_manifest() if r.part in ("edgar", "consumer", "simulated_scan")]
    if only:
        rows = [r for r in rows if r.doc_id in set(only)]
    rows = rows[:limit] if limit else rows
    vers = versions()
    log.info("versions %s", vers)

    total_pages = sum(int(r.pdf_pages or 0) for r in rows) * len(config_names)
    done_pages, spent, summary = 0, 0.0, {}
    for name in config_names:
        cfg = CONFIGS[name]
        converter = build_converter(cfg)
        stats = {"parsed": 0, "skipped": 0, "failed": [], "seconds": 0.0, "pages": 0}
        for row in rows:
            key = cache_key(row.source_sha256, vers["docling"], cfg)
            pages = int(row.pdf_pages or 0)
            if is_cached(cfg, row.doc_id, key):
                stats["skipped"] += 1
                total_pages -= pages
                continue
            try:
                payload = parse_one(converter, cfg, row, vers, key)
            except Exception as e:           # one bad document must not stop the run
                log.exception("parse_failed config=%s doc=%s", name, row.doc_id)
                stats["failed"].append({"doc_id": row.doc_id, "error": str(e)[:300]})
                continue
            write_atomic(cache_path(cfg, row.doc_id), payload)
            sec = payload["meta"]["seconds"]
            stats["parsed"] += 1
            stats["seconds"] += sec
            stats["pages"] += pages
            done_pages += pages
            spent += sec
            rate = spent / max(done_pages, 1)
            projected_h = rate * total_pages / 3600
            log.info("config=%s doc=%s pages=%d %.1fs (%.2f s/page) progress=%d/%d pages projected_total=%.2fh",
                     name, row.doc_id, pages, sec, sec / max(pages, 1), done_pages, total_pages, projected_h)
            if done_pages >= 50 and projected_h > max_hours:
                log.error("projected %.1fh exceeds cap %.1fh (D6): stopping to ask", projected_h, max_hours)
                summary[name] = stats
                summary["stopped_for_cap"] = True
                return summary
        stats["seconds"] = round(stats["seconds"], 1)
        summary[name] = stats
        log.info("config=%s done %s", name, json.dumps(stats))
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--configs", nargs="+", default=["P1", "P2"], choices=sorted(CONFIGS))
    ap.add_argument("--max-hours", type=float, default=8.0)
    ap.add_argument("--only", nargs="*", help="doc_ids to parse (default: whole manifest)")
    ap.add_argument("--limit", type=int, help="first N documents only (smoke test)")
    args = ap.parse_args()
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(CACHE_ROOT / "run.log")],
    )
    for noisy in ("docling", "docling_core", "docling_ibm_models", "transformers", "PIL"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    summary = run(args.configs, args.max_hours, args.only, args.limit)
    (CACHE_ROOT / "last_run_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 1 if summary.get("stopped_for_cap") else 0


if __name__ == "__main__":
    sys.exit(main())
