"""
Ingestion golden-set label schema and validator (scope v2 §5.4, F4).

One JSON file per page: evals/golden/ingest/pages/<page_id>.json

Coordinates: PDF points, origin TOP-LEFT, bbox = [x0, y0, x1, y1].
Boxes only need to be roughly right: matching uses IoU >= 0.5 (§8.1).

{
  "page_id": "deal_10_ca_p122", "doc_id": "deal_10_ca", "page": 122,
  "stratum": "table_heavy", "source_format": "html_rendered",
  "page_size": [612.0, 792.0],
  "status": "prefilled" | "verified",
  "from_scratch": false,             # true: labelled without seeing Docling output
  "prefill_source": "docling P1" | null,
  "labeled_by": null, "verified_on": null,
  "reference_text_source": "text_layer" | "source_page_text_layer" | "hand_typed",
  "reference_text": null,            # only when hand-typed (no usable text layer)
  "layout_elements": [{"id": "e1", "class": "section_header", "bbox": [...], "text": "..."}],
  "reading_order": ["e1", "e2", ...],
  "tables": [{"id": "t1", "element_id": "e7", "bbox": [...], "num_rows": 3, "num_cols": 2,
              "header_rows": [0],
              "cells": [{"row": 0, "col": 0, "row_span": 1, "col_span": 1, "text": "..."}]}],
  "key_values": [{"key": "Loan Amount", "value": "$162,000"}],
  "header_footer_lines": ["Page 3 of 5"],
  "notes": ""
}
"""
from __future__ import annotations

CLASSES = {
    "title", "section_header", "text", "list", "table",
    "page_header", "page_footer", "page_number", "other",
}
STRATA = {"digital_prose", "table_heavy", "scanned", "complex_layout"}
STATUSES = {"prefilled", "verified"}
REF_SOURCES = {"text_layer", "source_page_text_layer", "hand_typed"}

# Docling layout label -> §5.4 class. Recorded in the Decision Log.
DOCLING_TO_CLASS = {
    "title": "title",
    "section_header": "section_header",
    "text": "text",
    "paragraph": "text",
    "footnote": "text",
    "caption": "text",
    "list_item": "list",
    "table": "table",
    "document_index": "table",
    "page_header": "page_header",
    "page_footer": "page_footer",
}  # everything else (picture, checkbox_*, formula, code, key_value_region, form) -> "other"

REQUIRED = {"page_id", "doc_id", "page", "stratum", "page_size", "status", "from_scratch",
            "layout_elements", "reading_order", "tables", "key_values", "header_footer_lines",
            "reference_text_source"}
BBOX_TOLERANCE = 2.0     # points; rounding / anti-aliasing slack


def map_docling_label(label: str) -> str:
    return DOCLING_TO_CLASS.get(label, "other")


def _bbox_errors(where: str, bbox, w: float, h: float) -> list[str]:
    if not (isinstance(bbox, list) and len(bbox) == 4 and all(isinstance(v, (int, float)) for v in bbox)):
        return [f"{where}: bbox must be [x0, y0, x1, y1]"]
    x0, y0, x1, y1 = bbox
    errs = []
    if x0 >= x1 or y0 >= y1:
        errs.append(f"{where}: bbox has non-positive size {bbox}")
    t = BBOX_TOLERANCE
    if x0 < -t or y0 < -t or x1 > w + t or y1 > h + t:
        errs.append(f"{where}: bbox {bbox} outside page {w}x{h}")
    return errs


def _table_errors(t: dict, w: float, h: float) -> list[str]:
    tid = t.get("id", "?")
    errs = _bbox_errors(f"table {tid}", t.get("bbox"), w, h)
    nr, nc = t.get("num_rows"), t.get("num_cols")
    if not (isinstance(nr, int) and isinstance(nc, int) and nr > 0 and nc > 0):
        return errs + [f"table {tid}: num_rows/num_cols must be positive integers"]
    covered: dict[tuple[int, int], int] = {}
    for i, c in enumerate(t.get("cells", [])):
        r, col = c.get("row"), c.get("col")
        rs, cs = c.get("row_span", 1), c.get("col_span", 1)
        if not all(isinstance(v, int) for v in (r, col, rs, cs)) or rs < 1 or cs < 1:
            errs.append(f"table {tid} cell {i}: row/col/spans must be integers, spans >= 1")
            continue
        if r < 0 or col < 0 or r + rs > nr or col + cs > nc:
            errs.append(f"table {tid} cell {i}: ({r},{col}) span {rs}x{cs} outside {nr}x{nc} grid")
            continue
        for rr in range(r, r + rs):
            for cc in range(col, col + cs):
                if (rr, cc) in covered:
                    errs.append(f"table {tid}: grid position ({rr},{cc}) covered by cells "
                                f"{covered[(rr, cc)]} and {i} (overlapping spans)")
                covered[(rr, cc)] = i
    missing = [(r, c) for r in range(nr) for c in range(nc) if (r, c) not in covered]
    if missing:
        errs.append(f"table {tid}: {len(missing)} grid positions have no cell, e.g. {missing[:3]}")
    for hr in t.get("header_rows", []):
        if not (isinstance(hr, int) and 0 <= hr < nr):
            errs.append(f"table {tid}: header row {hr} outside 0..{nr - 1}")
    return errs


def validate_labels(lab: dict) -> list[str]:
    """Return a list of problems (empty = valid)."""
    missing = REQUIRED - lab.keys()
    if missing:
        return [f"missing keys: {sorted(missing)}"]
    errs: list[str] = []
    if lab["stratum"] not in STRATA:
        errs.append(f"stratum {lab['stratum']!r} not in {sorted(STRATA)}")
    if lab["status"] not in STATUSES:
        errs.append(f"status {lab['status']!r} not in {sorted(STATUSES)}")
    if lab["reference_text_source"] not in REF_SOURCES:
        errs.append(f"reference_text_source {lab['reference_text_source']!r} not in {sorted(REF_SOURCES)}")
    if lab["reference_text_source"] == "hand_typed" and not lab.get("reference_text"):
        errs.append("reference_text_source is hand_typed but reference_text is empty")
    if lab["status"] == "verified" and not lab.get("labeled_by"):
        errs.append("verified pages need labeled_by")

    w, h = lab["page_size"]
    ids = set()
    for e in lab["layout_elements"]:
        eid = e.get("id")
        if eid in ids:
            errs.append(f"duplicate element id {eid}")
        ids.add(eid)
        if e.get("class") not in CLASSES:
            errs.append(f"element {eid}: class {e.get('class')!r} not in {sorted(CLASSES)}")
        errs += _bbox_errors(f"element {eid}", e.get("bbox"), w, h)

    seen = set()
    for rid in lab["reading_order"]:
        if rid not in ids:
            errs.append(f"reading_order references missing element {rid}")
        if rid in seen:
            errs.append(f"reading_order lists {rid} twice")
        seen.add(rid)

    tids = set()
    for t in lab["tables"]:
        if t.get("id") in tids:
            errs.append(f"duplicate table id {t.get('id')}")
        tids.add(t.get("id"))
        if t.get("element_id") and t["element_id"] not in ids:
            errs.append(f"table {t.get('id')}: element_id {t['element_id']} not in layout_elements")
        errs += _table_errors(t, w, h)

    for i, kv in enumerate(lab["key_values"]):
        if not isinstance(kv, dict) or not kv.get("key"):
            errs.append(f"key_values[{i}]: needs a non-empty key")
    return errs
