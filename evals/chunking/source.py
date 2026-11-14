"""
Build a SourceDoc (cleaned page text + table row provenance) from a cached Docling parse.

Pipeline: parse cache -> production_adapter.to_extraction_result -> production cleaner
(version v2 by default) -> SourceDoc. Table rows are located in the cleaned page text in
order: Docling's page text has each table row as its non-empty cells joined by spaces
(PageView.text()); checked on the whole corpus 2026-11-12 (3,220 / 3,220 rows found).
"""
from __future__ import annotations

import logging

from evals.chunking.base import Element, SourceDoc, Span, TableSource, strip_blank
from evals.parsers.docling_doc import all_page_views
from evals.parsers.production_adapter import to_extraction_result

from app.pipeline.cleaner import clean_extraction  # noqa: E402  (path set by production_adapter)

log = logging.getLogger(__name__)
CLEANING_VERSION = "v2"


def _row_texts(table) -> dict[int, str]:
    """Raw (pre-clean) row text as Docling's page text writes it: non-empty cells, col order."""
    rows: dict[int, list[str]] = {}
    for c in sorted(table.cells, key=lambda c: (c.row, c.col)):
        if c.text:
            rows.setdefault(c.row, []).append(c.text)
    return {r: " ".join(v) for r, v in rows.items()}


def locate_elements(views: dict, pages: dict[int, str], tables: list[TableSource]) -> tuple[list[Element], int]:
    """
    Layout elements in reading order with their span in the cleaned page text. Page text is the
    elements' text joined by newlines (PageView.text), so each is found in order from a cursor.
    Tables map to SourceDoc.tables in the adapter's order (pages ascending, view order).
    Returns (elements, number of text elements not found).
    """
    out: list[Element] = []
    missing, t_idx = 0, 0
    for n in sorted(views):
        text, cursor = pages.get(n, ""), 0
        tables_here = {t["element_id"] for t in views[n].tables}
        for e in views[n].elements:
            if e["id"] in tables_here:
                ts = tables[t_idx]
                ends = [s.end for s in ts.row_spans if s]
                cursor = max([cursor] + ends)
                out.append(Element(n, "table", "", None, t_idx))
                t_idx += 1
                continue
            t = strip_blank(e["text"])
            if not t:
                continue
            at = text.find(t, cursor)
            span = None
            if at < 0:
                missing += 1
            else:
                span, cursor = Span(n, at, at + len(t)), at + len(t)
            out.append(Element(n, e["class"], t, span))
    return out, missing


def build_source(extraction, cleaning_version: str = CLEANING_VERSION, views: dict | None = None) -> SourceDoc:
    cleaned = clean_extraction(extraction, version=cleaning_version)
    pages = {p.page_number: p.text for p in cleaned.pages}
    tables: list[TableSource] = []
    cursor: dict[int, int] = {}
    missing = 0
    for i, (raw, ct) in enumerate(zip(extraction.tables, cleaned.tables)):
        text = pages.get(raw.page, "")
        pos = cursor.get(raw.page, 0)
        row_text = _row_texts(raw)
        spans: list[Span | None] = []
        for r in range(1, raw.rows + 1):
            s = row_text.get(r)
            at = text.find(s, pos) if s else -1
            if s and at < 0:
                missing += 1
            if at < 0:
                spans.append(None)
                continue
            core = strip_blank(s)                      # cell text may end in U+FEFF
            lead = s.find(core)
            spans.append(Span(raw.page, at + lead, at + lead + len(core)))
            pos = at + len(s)
        cursor[raw.page] = pos
        tables.append(TableSource(page=raw.page, index=i, markdown=ct.to_markdown(), rows=ct.data,
                                  row_spans=spans))
    if missing:
        log.warning("%s: %d table rows not found in page text", extraction.session_id, missing)
    fields = [(f.key, f.value) for f in cleaned.form_fields]
    src = SourceDoc(doc_id=extraction.session_id, pages=pages, tables=tables, form_fields=fields,
                    cleaned=cleaned)
    if views is not None:
        src.elements, lost = locate_elements(views, pages, tables)
        if lost:
            log.info("%s: %d layout elements not in cleaned page text", extraction.session_id, lost)
    return src


def load_source(cached: dict, doc_id: str, cleaning_version: str = CLEANING_VERSION) -> SourceDoc:
    """SourceDoc with layout elements (C4 needs them; other chunkers ignore them)."""
    return build_source(to_extraction_result(cached, doc_id), cleaning_version,
                        views=all_page_views(cached["document"]))
