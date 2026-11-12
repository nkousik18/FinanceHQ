"""
Cleaning and validation metrics (scope v2 §8.1), measured by running the production
cleaner and validator unchanged on Docling output (via evals/parsers/production_adapter).

What the production cleaner actually does (fastapi_service/app/pipeline/cleaner.py,
checked 2026-11-12; the scope's list of 7 fixes does not match the code):
  page text:   remove lines matching 4 hard-coded header patterns (^S$, Shinhan Bank, India, tagline)
  table cells: number-space repair, digit-letter split
  form fields: drop empty checkbox keys, number-space repair, digit-letter split,
               de-duplicate (key, page), flag truncated values
Repair counts below are per fix, so a fix that never fires shows 0.
Over-cleaning is measured two ways: page text (CER after vs before cleaning, per scope) and
table cells (number-space repair that changes a cell's numbers, e.g. "29 29" -> "2929";
Docling puts several values in one cell, which the Textract-era repair assumes can't happen).

The counters call the cleaner's own helpers so they can't drift from it.
"""
from __future__ import annotations

import time
from typing import Callable

from evals.ingest_metrics.normalize import numeric_tokens
from evals.ingest_metrics.text import cer
from evals.parsers.production_adapter import _SERVICE  # noqa: F401  (puts fastapi_service on sys.path)

from app.pipeline import cleaner as C  # noqa: E402
from app.pipeline.validator import validate_extraction  # noqa: E402

HEADER_PATTERN_NAMES = ["single_S", "shinhan_bank", "india", "tagline"]
OVERCLEAN_EPS = 1e-9


def _header_hits(text: str) -> dict[str, int]:
    hits = dict.fromkeys(HEADER_PATTERN_NAMES, 0)
    for line in text.split("\n"):
        for name, pat in zip(HEADER_PATTERN_NAMES, C._HEADER_PATTERNS):
            if pat.match(line.strip()):
                hits[name] += 1
                break
    return hits


def _value_repairs(raw: str) -> dict[str, int]:
    """Which value fixes fire on one value. `number_corrupted`: the number-space repair changed
    the value's numbers (e.g. "29 29" -> "2929"), not just its formatting ("1." -> "1")."""
    v = raw.strip()
    n = C._clean_number(v)
    d = C._clean_digit_letter_concat(n)
    corrupted = n != v and sorted(numeric_tokens(n)) != sorted(numeric_tokens(v))
    return {"number_space": int(n != v), "number_corrupted": int(corrupted), "digit_letter": int(d != n)}


def _chars(extraction_pages, tables_text: list[str]) -> int:
    return sum(len(p.text) for p in extraction_pages) + sum(len(t) for t in tables_text)


def run_doc(extraction, reference: Callable[[int], str | None] | None = None) -> dict:
    """
    Clean + validate one document and measure it.
    `reference(page)` -> reference text or None (for over-cleaning). Only called for pages the
    cleaner changed; pages it left alone count as checked and not over-cleaned.
    """
    t0 = time.perf_counter()
    cleaned = C.clean_extraction(extraction)
    t1 = time.perf_counter()
    report = validate_extraction(cleaned)
    t2 = time.perf_counter()

    repairs = {f"header:{k}": 0 for k in HEADER_PATTERN_NAMES}
    repairs.update({"table:number_space": 0, "table:number_corrupted": 0, "table:digit_letter": 0,
                    "field:checkbox_dropped": 0, "field:number_space": 0, "field:number_corrupted": 0,
                    "field:digit_letter": 0,
                    "field:deduplicated": 0, "field:truncated_flag": 0})
    for p in extraction.pages:
        for k, v in _header_hits(p.text).items():
            repairs[f"header:{k}"] += v
    for t in extraction.tables:
        for c in t.cells:
            for k, v in _value_repairs(c.text).items():
                repairs[f"table:{k}"] += v
    kept = [f for f in extraction.form_fields if f.value.strip() or f.key.strip() not in C._CHECKBOX_KEYS]
    repairs["field:checkbox_dropped"] = len(extraction.form_fields) - len(kept)
    for f in kept:
        for k, v in _value_repairs(f.value).items():
            repairs[f"field:{k}"] += v
    repairs["field:deduplicated"] = len(kept) - len(cleaned.form_fields)
    repairs["field:truncated_flag"] = sum(f.is_truncated for f in cleaned.form_fields)

    raw_cells = [c.text for t in extraction.tables for c in t.cells]
    clean_cells = [x for t in cleaned.tables for row in t.data for x in row]
    before = _chars(extraction.pages, raw_cells)
    after = _chars(cleaned.pages, clean_cells)

    over, checked = [], 0
    raw_by_page = {p.page_number: p.text for p in extraction.pages}
    for cp in cleaned.pages:
        raw = raw_by_page[cp.page_number]
        if cp.text == raw.strip():
            checked += 1                  # cleaner didn't touch this page's text
            continue
        ref = reference(cp.page_number) if reference else None
        if ref is None:
            continue
        checked += 1
        cb, ca = cer(raw, ref), cer(cp.text, ref)
        if cb is not None and ca is not None and ca > cb + OVERCLEAN_EPS:
            over.append(cp.page_number)

    codes: dict[str, int] = {}
    for i in report.issues:
        key = f"{i.severity.value}:{i.code}"
        codes[key] = codes.get(key, 0) + 1
    return {
        "pages": len(extraction.pages),
        "repairs": repairs,
        "chars_before": before,
        "chars_removed": before - after,
        "chars_removed_rate": (before - after) / before if before else None,
        "overclean_pages": over,
        "table_cells": len(raw_cells),
        "overclean_checked_pages": checked,
        "validation": {"passed": report.passed, "errors": len(report.errors), "warnings": len(report.warnings),
                       "codes": codes, "overall_confidence": report.overall_confidence},
        "latency_ms": {"clean": 1000 * (t1 - t0), "validate": 1000 * (t2 - t1)},
    }


def aggregate(docs: list[dict]) -> dict:
    """Sum repair counts and validation codes; rates over pages/documents."""
    if not docs:
        return {"documents": 0}
    repairs: dict[str, int] = {}
    codes: dict[str, int] = {}
    docs_with_code: dict[str, int] = {}
    for d in docs:
        for k, v in d["repairs"].items():
            repairs[k] = repairs.get(k, 0) + v
        for k, v in d["validation"]["codes"].items():
            codes[k] = codes.get(k, 0) + v
            docs_with_code[k] = docs_with_code.get(k, 0) + 1
    before = sum(d["chars_before"] for d in docs)
    checked = sum(d["overclean_checked_pages"] for d in docs)
    clean_ms = sorted(d["latency_ms"]["clean"] for d in docs)
    val_ms = sorted(d["latency_ms"]["validate"] for d in docs)
    pct = lambda s, q: s[min(len(s) - 1, int(round(q * (len(s) - 1))))]
    return {
        "documents": len(docs),
        "pages": sum(d["pages"] for d in docs),
        "repairs": repairs,
        "chars_removed_rate": sum(d["chars_removed"] for d in docs) / before if before else None,
        "overclean_rate": (sum(len(d["overclean_pages"]) for d in docs) / checked) if checked else None,
        "overclean_checked_pages": checked,
        "table_cells": sum(d["table_cells"] for d in docs),
        "table_cell_corruption_rate": (repairs.get("table:number_corrupted", 0) / sum(d["table_cells"] for d in docs)
                                       if sum(d["table_cells"] for d in docs) else None),
        "validation_failed_docs": sum(not d["validation"]["passed"] for d in docs),
        "validation_codes": codes,
        "docs_with_code": docs_with_code,
        "latency_ms": {"clean_p50": pct(clean_ms, .5), "clean_p95": pct(clean_ms, .95),
                       "validate_p50": pct(val_ms, .5), "validate_p95": pct(val_ms, .95)},
    }
