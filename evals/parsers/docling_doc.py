"""
Read a cached Docling document as simple per-page views (app venv; no Docling import).

Shared by the golden-set pre-fill and the §8.1 metrics, so both read Docling
output the same way. Coordinates are converted to PDF points, top-left origin,
[x0, y0, x1, y1], matching the golden labels.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from evals.golden.ingest.schema import map_docling_label

_PAGE_NUM_RE = re.compile(r"^\s*(?:page\s+)?[-–]?\s*[0-9ivxlc]+\s*[-–]?(?:\s+of\s+\d+)?\s*$", re.IGNORECASE)
HEADER_FOOTER = ("page_header", "page_footer", "page_number")


def _resolve(doc: dict, ref: str) -> dict:
    _, coll, idx = ref.split("/")
    return doc[coll][int(idx)]


def reading_order_refs(doc: dict) -> list[str]:
    """Leaf item refs (texts/tables/pictures) in Docling's body order, then furniture, then any left over.
    Texts that are children of a table/picture are dropped (their content is in the parent)."""
    out: list[str] = []
    seen: set[str] = set()

    def walk(node: dict) -> None:
        for child in node.get("children", []):
            ref = child["$ref"]
            if ref in seen:
                continue
            seen.add(ref)
            item = _resolve(doc, ref)
            if ref.startswith("#/groups/"):
                walk(item)
            else:
                out.append(ref)
                if ref.startswith(("#/tables/", "#/pictures/")):
                    walk(item)

    walk(doc["body"])
    walk(doc.get("furniture", {}))
    for coll in ("texts", "tables", "pictures"):
        for item in doc.get(coll, []):
            if item["self_ref"] not in seen:
                out.append(item["self_ref"])
    parents = {ch["$ref"] for coll in ("tables", "pictures") for item in doc.get(coll, [])
               for ch in item.get("children", [])}
    return [r for r in out if r not in parents]


def to_topleft(bbox: dict, page_h: float) -> list[float]:
    l, t, r, b = bbox["l"], bbox["t"], bbox["r"], bbox["b"]
    if bbox.get("coord_origin", "BOTTOMLEFT") == "BOTTOMLEFT":
        t, b = page_h - t, page_h - b
    return [round(min(l, r), 1), round(min(t, b), 1), round(max(l, r), 1), round(max(t, b), 1)]


def table_grid(item: dict) -> dict:
    """Docling table -> {num_rows, num_cols, header_rows, cells} with a complete grid
    (Docling omits empty cells; they are filled as empty 1x1 cells)."""
    data = item["data"]
    nr, nc = data["num_rows"], data["num_cols"]
    cells, covered, header_rows = [], set(), set()
    for c in data["table_cells"]:
        r, col = c["start_row_offset_idx"], c["start_col_offset_idx"]
        rs = max(1, c["end_row_offset_idx"] - r)
        cs = max(1, c["end_col_offset_idx"] - col)
        cells.append({"row": r, "col": col, "row_span": rs, "col_span": cs, "text": c.get("text", "")})
        covered |= {(rr, cc) for rr in range(r, r + rs) for cc in range(col, col + cs)}
        if c.get("column_header"):
            header_rows.add(r)
    for r in range(nr):
        for col in range(nc):
            if (r, col) not in covered:
                cells.append({"row": r, "col": col, "row_span": 1, "col_span": 1, "text": ""})
    cells.sort(key=lambda c: (c["row"], c["col"]))
    return {"num_rows": nr, "num_cols": nc, "header_rows": sorted(header_rows), "cells": cells}


@dataclass
class PageView:
    page: int
    size: tuple[float, float]
    elements: list[dict] = field(default_factory=list)   # {id, class, bbox, text, ref}
    tables: list[dict] = field(default_factory=list)     # {id, element_id, bbox, num_rows, num_cols, header_rows, cells}
    key_values: list[dict] = field(default_factory=list)  # {key, value}

    @property
    def reading_order(self) -> list[str]:
        return [e["id"] for e in self.elements]

    @property
    def header_footer_lines(self) -> list[str]:
        return [e["text"].strip() for e in self.elements if e["class"] in HEADER_FOOTER and e["text"].strip()]

    def text(self) -> str:
        """All text on the page in Docling's reading order; tables row by row."""
        tables = {t["element_id"]: t for t in self.tables}
        parts = []
        for e in self.elements:
            t = tables.get(e["id"])
            if t:
                rows: dict[int, list[str]] = {}
                for c in t["cells"]:
                    if c["text"]:
                        rows.setdefault(c["row"], []).append(c["text"])
                parts.extend(" ".join(v) for _, v in sorted(rows.items()))
            elif e["text"]:
                parts.append(e["text"])
        return "\n".join(parts)


def all_page_views(doc: dict) -> dict[int, PageView]:
    """Every page's view in one pass (calling page_view per page re-walks the whole document)."""
    views = {int(k): PageView(page=int(k), size=(m["size"]["width"], m["size"]["height"]))
             for k, m in doc.get("pages", {}).items()}
    for ref in reading_order_refs(doc):
        item = _resolve(doc, ref)
        seen_pages: set[int] = set()
        for prov in item.get("prov", []):
            if prov["page_no"] in views and prov["page_no"] not in seen_pages:
                seen_pages.add(prov["page_no"])
                _add_item(views[prov["page_no"]], item, ref, prov)
    # key_values stay empty: Docling's key_value_items is empty on these flat PDFs
    # (checked 2026-11-04). If a later parser fills them, map them here.
    return views


def page_view(doc: dict, page: int) -> PageView:
    return all_page_views(doc)[page]


def _add_item(view: PageView, item: dict, ref: str, prov: dict) -> None:
    """Append one Docling item to a page view, using its first box on that page."""
    bbox = to_topleft(prov["bbox"], view.size[1])
    cls = map_docling_label(item.get("label", ""))
    text = item.get("text", "") or ""
    marker = (item.get("marker") or "").strip()
    if marker and not text.startswith(marker):
        text = f"{marker} {text}"          # Docling keeps list numbers ("3.", "(a)") in a separate field
    if cls == "page_footer" and _PAGE_NUM_RE.match(text):
        cls = "page_number"
    eid = f"e{len(view.elements) + 1}"
    view.elements.append({"id": eid, "class": cls, "bbox": bbox, "text": text, "ref": ref})
    if ref.startswith("#/tables/"):
        view.tables.append({"id": f"t{len(view.tables) + 1}", "element_id": eid, "bbox": bbox,
                            **table_grid(item)})


def page_count(doc: dict) -> int:
    return len(doc.get("pages", {}))
