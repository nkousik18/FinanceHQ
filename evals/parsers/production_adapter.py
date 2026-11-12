"""
Docling parse cache -> production `ExtractionResult` (fastapi_service/app/pipeline/extractor.py).

Lets the production cleaner, validator (R1 metrics) and chunker (R2) run unchanged on
Docling output. Mirrors how the Textract extractor fills each field:

| ExtractionResult field | Textract (production)                    | Docling (this adapter)                              |
|---|---|---|
| pages                  | only pages with LINE/WORD blocks          | only pages with at least one word (blank pages dropped, like Textract) |
| page.text              | LINE blocks, one per line (tables incl.)  | PageView.text(): elements in reading order, tables row by row |
| page.word_count        | WORD blocks                               | whitespace words in page.text                        |
| page.avg_confidence    | mean WORD confidence (0-100)              | word-weighted: OCR words = Tesseract/Vision conf x 100, text-layer words = 100 (no OCR uncertainty) |
| low_confidence_words   | words < 80                                | OCR words < 0.80 (text-layer words never count)      |
| has_handwriting        | TextType == HANDWRITING                   | False (Docling has no handwriting flag)              |
| tables                 | TABLE/CELL, 1-based, merged cells repeat  | Docling grid, 1-based, text on the anchor cell only  |
| form_fields            | KEY_VALUE_SET                             | [] (Docling finds none on these flat PDFs)           |
| full_text              | pages + "[TABLE]" blocks, page breaks     | same assembly                                        |
| overall_avg_confidence | word-weighted mean over pages             | same                                                 |

The confidence choice for text-layer words is recorded in the scope Decision Log (2026-11-12).
"""
from __future__ import annotations

import sys
from pathlib import Path

from evals.parsers.docling_doc import PageView, all_page_views

_SERVICE = Path(__file__).resolve().parents[2] / "fastapi_service"
if str(_SERVICE) not in sys.path:
    sys.path.insert(0, str(_SERVICE))

from app.pipeline.extractor import ExtractionResult, PageResult, Table, TableCell  # noqa: E402

TEXT_LAYER_CONF = 100.0     # text-layer words: exact characters from the PDF
LOW_WORD_CONF = 0.80        # production counts words below 80 (0-100 scale)


def _page_confidence(n_words: int, ocr_confs: list[float]) -> tuple[float, int]:
    """(avg confidence 0-100, low-confidence word count) for one page."""
    n_ocr = len(ocr_confs)
    n_layer = max(n_words - n_ocr, 0)
    total = n_layer + n_ocr
    if total == 0:
        return 0.0, 0
    avg = (n_layer * TEXT_LAYER_CONF + sum(100.0 * c for c in ocr_confs)) / total
    return round(avg, 2), sum(c < LOW_WORD_CONF for c in ocr_confs)


def _table(view_table: dict, page: int) -> Table:
    cells = [TableCell(row=c["row"] + 1, col=c["col"] + 1, text=c["text"], confidence=TEXT_LAYER_CONF)
             for c in view_table["cells"]]
    return Table(page=page, rows=view_table["num_rows"], cols=view_table["num_cols"], cells=cells)


def page_result(view: PageView, ocr_confs: list[float]) -> PageResult:
    text = view.text()
    n_words = len(text.split())
    avg, low = _page_confidence(n_words, ocr_confs)
    return PageResult(page_number=view.page, text=text, word_count=n_words, avg_confidence=avg,
                      low_confidence_words=low, has_handwriting=False)


def to_extraction_result(cached: dict, session_id: str) -> ExtractionResult:
    """Build an ExtractionResult from one cached Docling parse (`load_cached(...)` output)."""
    views = all_page_views(cached["document"])
    ocr = cached.get("ocr_cells", {})
    pages, tables = [], []
    for n in sorted(views):
        confs = [c["confidence"] for c in ocr.get(str(n), []) if c.get("text", "").strip()]
        pr = page_result(views[n], confs)
        if pr.word_count:                  # Textract emits no PageResult for a page without words
            pages.append(pr)
        tables += [_table(t, n) for t in views[n].tables]

    by_page: dict[int, list[str]] = {}
    for t in tables:
        by_page.setdefault(t.page, []).append(t.to_text())
    sections = []
    for p in pages:
        s = p.text
        if p.page_number in by_page:
            s += "\n\n[TABLE]\n" + "\n\n[TABLE]\n".join(by_page[p.page_number])
        sections.append(s)

    total_words = sum(p.word_count for p in pages)
    overall = sum(p.avg_confidence * p.word_count for p in pages) / total_words if total_words else 0.0
    return ExtractionResult(
        session_id=session_id,
        full_text="\n\n--- Page Break ---\n\n".join(sections),
        pages=pages, tables=tables, form_fields=[],
        total_pages=len(pages),
        overall_avg_confidence=round(overall, 2),
        has_handwriting=False,
    )
